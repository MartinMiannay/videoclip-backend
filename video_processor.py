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
SUB_FONT = "Montserrat-Bold"
SUB_FONTSIZE = 52
SUB_BORDER_W = 2          # thin black outline
SUB_BOX_BORDER = 8        # padding around subtitle background box (px)
SUB_Y_RATIO = 0.72        # vertical position as fraction of frame height (classic)
SUB_Y_KEO = 0.80          # vertical position for "keo" style (80% from top)
SUB_Y_TOVARITCH = 0.40    # vertical position for "tovaritch" style (40% from top)
SUB_SHIFT_MS = 0.0        # no pre-display shift — eliminates card-boundary overlap
SUB_SILENCE_GAP = 0.3     # gap in seconds that means silence (no card)
SUB_MIN_CARD_GAP = 0.05   # minimum gap enforced between consecutive subtitle cards

_FILLER_WORDS = {"hm", "hum", "euh", "eh", "mmm", "ah"}  # isolated filler sounds to suppress

# Keo style: French function words rendered at lighter weight
_KEO_FILLER_WORDS_FR = frozenset({
    "le", "la", "les", "de", "du", "un", "une", "que", "qui",
    "et", "en", "à", "au", "par", "sur", "pour", "pas", "ne",
})

# Set False to skip all drawtext rendering (subtitles + title card).
# Disable while debugging FFmpeg filter parse errors; re-enable once rendering works.
SUBTITLES_ENABLED = True

# Face detection thresholds
FACE_CONF_THRESHOLD = 0.7

# Speaker framing
FACE_Y_TARGET_RATIO = 0.25   # target face position from top of output frame (25% = upper body visible)
FACE_CROP_H_RATIO = 0.80     # crop height as fraction of source_h — zoom in to allow Y adjustment
SPEAKER_SWITCH_MIN_INTERVAL = 2.0  # minimum seconds between camera switches

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
    # Prefer Montserrat-Bold; fall back to Arial Bold if not installed.
    _montserrat_src = Path("C:/Windows/Fonts/Montserrat-Bold.ttf")
    _arial_src      = Path("C:/Windows/Fonts/arialbd.ttf")
    _win_font_src   = _montserrat_src if _montserrat_src.exists() else _arial_src
    _win_font_name  = _win_font_src.name
    _win_font_dst   = ASSETS_DIR / _win_font_name
    if not _win_font_dst.exists() and _win_font_src.exists():
        import shutil as _shutil
        ASSETS_DIR.mkdir(exist_ok=True)
        _shutil.copy2(_win_font_src, _win_font_dst)
    FONT_PATH = f"assets/{_win_font_name}"  # relative to ROOT_DIR, passed as ffmpeg cwd

    # Arial Regular (keo filler words — lighter weight)
    _arial_regular_src = Path("C:/Windows/Fonts/arial.ttf")
    _arial_regular_dst = ASSETS_DIR / "arial.ttf"
    if not _arial_regular_dst.exists() and _arial_regular_src.exists():
        import shutil as _shutil2; ASSETS_DIR.mkdir(exist_ok=True)
        _shutil2.copy2(_arial_regular_src, _arial_regular_dst)
    FONT_PATH_REGULAR = "assets/arial.ttf" if _arial_regular_dst.exists() else FONT_PATH
    FONT_PATH_BOLD = FONT_PATH  # Arial Bold / Montserrat Bold already the primary

    # Impact (tovaritch style)
    _impact_src = Path("C:/Windows/Fonts/impact.ttf")
    _impact_dst = ASSETS_DIR / "impact.ttf"
    if not _impact_dst.exists() and _impact_src.exists():
        import shutil as _shutil3; ASSETS_DIR.mkdir(exist_ok=True)
        _shutil3.copy2(_impact_src, _impact_dst)
    FONT_PATH_IMPACT = "assets/impact.ttf" if _impact_dst.exists() else FONT_PATH
else:
    FONT_PATH = "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf"
    FONT_PATH_BOLD = FONT_PATH
    # Liberation Sans Regular is a metric-compatible Arial substitute, included in most distros
    _liberation_regular = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    _arial_regular_linux = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
    if Path(_liberation_regular).exists():
        FONT_PATH_REGULAR = _liberation_regular
    elif Path(_arial_regular_linux).exists():
        FONT_PATH_REGULAR = _arial_regular_linux
    else:
        FONT_PATH_REGULAR = FONT_PATH  # fall back to bold — all words bold
    _impact_linux = "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"
    FONT_PATH_IMPACT = _impact_linux if Path(_impact_linux).exists() else FONT_PATH

CLIP_MIN_DURATION = 20   # seconds
CLIP_MAX_DURATION = 120  # seconds

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
- Split aggressively around moments of high tension: shocking stats, contrarian \
  claims, personal sacrifice stories, direct challenges, or dramatic revelations. \
  These are prime clip candidates and must not be buried inside a larger segment.
- You MUST distribute themes evenly across the ENTIRE video. Divide the video into \
  4 equal quarters and find at least 3–4 themes in EACH quarter. Never cluster more \
  than 4 themes in any single quarter.

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

SEGMENT_FALLBACK_PROMPT = """\
You are an expert content analyst for French business and entrepreneurship interviews.

The transcript excerpt below covers {start:.0f}s–{end:.0f}s of an interview. \
This section was missed in the initial segmentation. \
Find 3–4 distinct themes within this section.

Rules:
- Return ONLY themes whose start/end fall within {start:.0f}s–{end:.0f}s.
- Each theme is one topic, story, opinion, or standalone moment.
- Themes can be as short as 30 seconds. Do NOT merge adjacent distinct topics.
- Provide a short French label (≤ 8 words) and a one-sentence description.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no commentary:
{{
  "themes": [
    {{
      "start": <float seconds>,
      "end": <float seconds>,
      "theme": "<short French label>",
      "description": "<one sentence>"
    }}
  ]
}}
"""

BOUNDARY_PROMPT = """\
You are a precision video editor for French short-form business content.

You will receive transcript excerpts for ALL themes from an interview. \
Select the best 12–14 clips total — one per theme, choosing the themes with \
the strongest viral potential. You MUST return at least 12 clips, never fewer \
than 10. Every selected theme MUST have a clip; do not skip a theme you have chosen.

HOOK SELECTION — for each theme, scan ALL sentences in the transcript excerpt \
and find the single most shocking, controversial or emotionally charged sentence. \
That sentence must be the hook start.

STRONG hooks contain: a specific number, a personal confession, a contradiction, \
a direct challenge, or a surprising consequence.

WEAK hooks are: context-setting, topic introductions, questions without tension, \
transitions.

If the strongest line is mid-excerpt, start there even if it cuts off context. \
The viewer can figure it out.

START RULES — the clip must open on:
- A surprising stat, bold claim, provocative question, or mid-tension moment
- The exact word where the tension or insight begins
- Start mid-conflict or mid-revelation, NEVER on context-setting
- Prefer clips where the first sentence creates immediate dissonance or tension
- The hook must make the viewer think "wait, what?" in the first 3 seconds
- NEVER start on "donc", "voilà", "en fait", "bah", "alors", "bonjour", \
  "aujourd'hui je vais", or any other filler or scene-setting phrase

END RULES — cut immediately after:
- The key insight or punchline lands
- Before the speaker pivots to explanation, context, or a new point

DURATION: Aim for 30–90 seconds. If the best moment for a theme is slightly \
outside that range, include it anyway — always return the best clip for each \
selected theme even if imperfect.

VIRAL HOOK EXAMPLES (real high-performing French business clips):
- "Tu n'es pas intelligent si c'est toi qui crée la valeur" — contrarian reframe \
  of entrepreneurship
- "Tu penses que tu es libre, mais tu es 100% esclave" — shocking reframe
- "1000€ de bénéfice valent en réalité 5000 ou 7000€" — shocking stat
- "Je dormais dans mes bureaux, mes employés se lavaient dans ma douche" — \
  personal sacrifice story
- "T'as 20 ans, y'a pas besoin de partir en vacances" — direct challenge
- "Je ne pouvais pas survivre avec un loyer de 15 000€" — dramatic consequence
- "Tu vas savoir si tu es un entrepreneur ou pas" — direct challenge/tension
- "Embaucher ma femme c'est une bonne idée ?"
- "Un téléphone pro ? Seulement quand on gagne 1 000 000 €"
- "Pourquoi faire du black vous appauvrit ?"
- "Il fait 500 000€/an et voyage pendant 5 mois !"

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
    speaker: str | None = None  # WhisperX diarization label e.g. "SPEAKER_00"


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
    words: list[Word]          # words remapped to output timeline (t=0 at clip start)
    subtitle_cards: list[SubtitleCard] = field(default_factory=list)
    # keep_segments in source-video absolute time; empty means use start/end directly
    segments: list[tuple[float, float]] = field(default_factory=list)

    @property
    def output_duration(self) -> float:
        if self.segments:
            return sum(e - s for s, e in self.segments)
        return self.end - self.start


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
    compute_type = "float16"

    logger.info("Loading WhisperX large-v3 model on %s (compute_type=%s)…", device, compute_type)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

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

    # Optional speaker diarization — requires HF_TOKEN and pyannote
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            logger.info("Running speaker diarization (HF_TOKEN found)…")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segments = diarize_model(audio_path)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info("Diarization complete")
        except Exception as _dia_err:
            logger.warning("Diarization failed (%s) — speaker switching will use pause detection", _dia_err)

    words: list[Word] = []
    for segment in result["segments"]:
        seg_speaker: str | None = segment.get("speaker")
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
                speaker=w.get("speaker") or seg_speaker,
            ))

    n_speakers = len({w.speaker for w in words if w.speaker})
    logger.info("Transcription complete — %d words, %d speaker(s) detected", len(words), n_speakers)
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
        output_words = [
            Word(text=w.text, start=w.start - clip_start, end=w.end - clip_start, precise=w.precise)
            for w in grp
        ]
        raw_start = output_words[0].start
        raw_end = output_words[-1].end
        cards.append(SubtitleCard(
            words=output_words,
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

    # Pass 1: push each card's display_start forward if it would overlap the previous card.
    for i in range(1, len(cards)):
        min_start = cards[i - 1].display_end + SUB_MIN_CARD_GAP
        if cards[i].display_start < min_start:
            cards[i].display_start = min_start

    # Pass 2: also clamp each card's display_end so it never reaches the next card.
    # This prevents any between(t,X,Y) window from overlapping the next card's window,
    # which is the root cause of duplicate-word glitches at card boundaries.
    for i in range(len(cards) - 1):
        max_end = cards[i + 1].display_start - SUB_MIN_CARD_GAP
        if cards[i].display_end > max_end:
            cards[i].display_end = max(cards[i].display_start + 0.05, max_end)

    return cards


# ---------------------------------------------------------------------------
# 3. Claude clip selection pipeline (5 steps)
# ---------------------------------------------------------------------------

def _claude_json(system: str, user: str, label: str, sleep_seconds: int = 60, max_tokens: int = 4096) -> Any:
    """Single Claude API call → parsed JSON. Sleeps before the call to avoid rate limits.
    Retries once after 30 seconds if the response is empty or invalid JSON."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(2):
        logger.info("Claude [%s] sleeping %ds before request (attempt %d)…", label, sleep_seconds, attempt + 1)
        time.sleep(sleep_seconds)
        logger.info("Claude [%s] sending request — user msg %d chars", label, len(user))
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = message.content[0].text.strip()
        logger.info(
            "Claude [%s] raw response (%d chars, stop_reason=%s): %.500s",
            label, len(raw), message.stop_reason, raw,
        )
        if not raw:
            logger.error("Claude [%s] empty response (attempt %d)", label, attempt + 1)
            if attempt == 0:
                logger.info("Claude [%s] retrying after 30s…", label)
                time.sleep(30)
                continue
            raise ValueError(f"Claude [{label}] empty response after retry")
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        try:
            result = json.loads(raw)
            logger.info("Claude [%s] JSON parse OK", label)
            return result
        except json.JSONDecodeError as exc:
            logger.error("Claude [%s] JSON parse FAILED (attempt %d): %s\nFull raw response:\n%s", label, attempt + 1, exc, raw)
            # Try to salvage complete objects from a truncated array response
            salvaged = _salvage_partial_json_list(raw, label)
            if salvaged is not None:
                return salvaged
            if attempt == 0:
                logger.info("Claude [%s] retrying after 30s…", label)
                time.sleep(30)
                continue
            raise ValueError(f"Claude [{label}] invalid JSON after retry: {exc}") from exc

    raise RuntimeError(f"Claude [{label}] unreachable")


def _salvage_partial_json_list(raw: str, label: str) -> Any | None:
    """
    If Claude's response was cut off mid-array, extract whatever complete
    top-level objects were returned and wrap them back in the expected envelope.
    Returns a dict with the salvaged list, or None if nothing could be recovered.
    """
    # Find the top-level key and its array (e.g. "clips" or "themes")
    key_match = re.search(r'"(clips|themes)"\s*:\s*\[', raw)
    if not key_match:
        return None
    key = key_match.group(1)
    array_start = key_match.end() - 1  # points at '['

    # Walk the raw string collecting complete {...} objects
    salvaged_items: list[Any] = []
    i = array_start + 1
    depth = 0
    obj_start: int | None = None
    while i < len(raw):
        ch = raw[i]
        if ch == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    salvaged_items.append(json.loads(raw[obj_start:i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = None
        i += 1

    if salvaged_items:
        logger.warning("Claude [%s] salvaged %d complete objects from truncated response", label, len(salvaged_items))
        return {key: salvaged_items}
    return None


def segment_themes_with_claude(words: list[Word]) -> list[dict[str, Any]]:
    """
    Step 1 — Thematic segmentation.
    Returns [{start, end, theme, description}, ...] (15–20 entries).
    Forces even distribution across all quarters; calls Claude again for any empty quarter.
    """
    thinned_words = words[::2]  # every other word — enough for theme detection
    transcript = words_to_transcript_text(thinned_words)
    video_end = words[-1].end if words else 0.0
    logger.info(
        "Step 1 — thematic segmentation — transcript: %d words (thinned from %d), %d chars, video end %.1fs",
        len(thinned_words), len(words), len(transcript), video_end,
    )
    data = _claude_json(SEGMENT_PROMPT, transcript, "segmentation", sleep_seconds=60)
    themes = data.get("themes", [])

    themes = _log_and_fix_quarter_distribution(themes, words, video_end)

    if len(themes) < 15:
        logger.warning(
            "  !! Only %d themes returned (expected ≥ 15). Consider re-running.", len(themes)
        )
    return themes


def _log_and_fix_quarter_distribution(
    themes: list[dict[str, Any]],
    words: list[Word],
    video_end: float,
) -> list[dict[str, Any]]:
    """Log theme counts per quarter; call Claude for any quarter with 0 themes."""
    quarter = video_end / 4
    boundaries = [(i * quarter, (i + 1) * quarter) for i in range(4)]

    def quarter_index(t: dict[str, Any]) -> int:
        mid = (t.get("start", 0) + t.get("end", 0)) / 2
        for i, (q_start, q_end) in enumerate(boundaries):
            if q_start <= mid < q_end:
                return i
        return 3  # clamp last theme to Q4

    counts = [0, 0, 0, 0]
    for t in themes:
        counts[quarter_index(t)] += 1

    logger.info(
        "  Theme distribution across quarters (each ~%.0fs):\n"
        "    Q1 [%.0fs–%.0fs]: %d themes\n"
        "    Q2 [%.0fs–%.0fs]: %d themes\n"
        "    Q3 [%.0fs–%.0fs]: %d themes\n"
        "    Q4 [%.0fs–%.0fs]: %d themes",
        quarter,
        boundaries[0][0], boundaries[0][1], counts[0],
        boundaries[1][0], boundaries[1][1], counts[1],
        boundaries[2][0], boundaries[2][1], counts[2],
        boundaries[3][0], boundaries[3][1], counts[3],
    )
    logger.info(
        "  All themes:%s",
        "".join(
            f"\n    [{i+1:2d}] Q{quarter_index(t)+1} {t.get('theme', '?')!r:50s} "
            f"{t.get('start', 0):.1f}s–{t.get('end', 0):.1f}s "
            f"({t.get('end', 0) - t.get('start', 0):.0f}s)"
            for i, t in enumerate(themes)
        ),
    )

    for qi, (q_start, q_end) in enumerate(boundaries):
        if counts[qi] > 0:
            continue
        logger.warning(
            "  !! Q%d [%.0fs–%.0fs] has 0 themes — running fallback segmentation…",
            qi + 1, q_start, q_end,
        )
        section_words = [w for w in words if q_start <= w.start < q_end]
        thinned = section_words[::2]
        if not thinned:
            logger.warning("  !! Q%d: no words found in section, skipping fallback", qi + 1)
            continue
        section_transcript = words_to_transcript_text(thinned)
        prompt = SEGMENT_FALLBACK_PROMPT.format(start=q_start, end=q_end)
        fallback_data = _claude_json(prompt, section_transcript, f"segmentation-fallback-Q{qi+1}", sleep_seconds=30)
        fallback_themes = fallback_data.get("themes", [])
        logger.info("  Fallback Q%d: got %d themes", qi + 1, len(fallback_themes))
        themes = themes + fallback_themes

    themes.sort(key=lambda t: t.get("start", 0))
    return themes


def find_clip_boundaries_with_claude(
    themes: list[dict[str, Any]],
    words: list[Word],
) -> list[dict[str, Any]]:
    """
    Step 2 — Send all themes to Claude in one call and ask for the best
    clips ranked by virality. Returns all valid clips after duration
    validation and hook-snapping to the first spoken word.
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

    data = _claude_json(BOUNDARY_PROMPT, user_msg, "boundaries", sleep_seconds=30, max_tokens=8192)
    all_raw_clips = data.get("clips", [])

    # Enforce one clip per theme at the code level
    seen_themes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in all_raw_clips:
        theme_label = c.get("theme", "")
        if theme_label in seen_themes:
            logger.info(
                "  Clip '%s' dropped — theme '%s' already has a clip",
                c.get("title", "?"), theme_label,
            )
            continue
        seen_themes.add(theme_label)
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

    logger.info(
        "  Clip pipeline: %d raw → %d after dedup → %d after validation",
        len(all_raw_clips), len(deduped), len(valid),
    )

    valid.sort(key=lambda c: float(c.get("virality_score", 0)), reverse=True)
    logger.info(
        "  Final %d clips: %s",
        len(valid),
        [f'"{c.get("title", "?")}" score={c.get("virality_score", "?")} ({float(c["end"]) - float(c["start"]):.0f}s)' for c in valid],
    )
    return valid


# Silence trimming ────────────────────────────────────────────────────────────

_SILENCE_TRIM_THRESHOLD = 0.8   # seconds

_TRIM_FILLER_WORDS = {"euh", "heum", "hm", "hmm", "mmm", "bah", "ben", "voilà"}


def _remap_words_to_output(
    words: list[Word],
    segments: list[tuple[float, float]],
) -> list[Word]:
    """
    Given words with source-timeline timestamps and a list of keep segments
    (in source time), return new Word objects whose timestamps are in the
    output timeline (gaps removed, output starts at t=0).
    """
    remapped: list[Word] = []
    output_offset = 0.0
    for seg_start, seg_end in segments:
        for w in words:
            if seg_start <= w.start and w.end <= seg_end + 0.05:
                remapped.append(Word(
                    text=w.text,
                    start=output_offset + (w.start - seg_start),
                    end=output_offset + (w.end - seg_start),
                    precise=w.precise,
                    speaker=w.speaker,
                ))
        output_offset += seg_end - seg_start
    return remapped


def trim_clip_silences(
    clips: list[dict[str, Any]],
    words: list[Word],
) -> list[dict[str, Any]]:
    """
    Step 4 — Internal silence trimming (no API call).
    For each clip:
    - Strip isolated filler words ("euh", "hm", etc.) from the word list.
    - Advance start to the first non-filler word if leading silence > threshold.
    - Pull end back to the last non-filler word's end if trailing silence > threshold.
    - Log exactly how many seconds are removed at each end.
    Clips that become too short after trimming are dropped.
    """
    logger.info("Silence trimming — %d clips (threshold=%.1fs, fillers=%s)…",
                len(clips), _SILENCE_TRIM_THRESHOLD, sorted(_TRIM_FILLER_WORDS))
    trimmed: list[dict[str, Any]] = []

    for c in clips:
        title = c.get("title", "?")
        start = float(c["start"])
        end = float(c["end"])

        # Build word list for this clip, stripping isolated filler words
        all_clip_words = [w for w in words if start <= w.start <= end]
        clip_words = [
            w for w in all_clip_words
            if w.text.lower().strip(".,!?…") not in _TRIM_FILLER_WORDS
        ]
        n_fillers = len(all_clip_words) - len(clip_words)
        if n_fillers:
            logger.info("  Clip '%s' — removed %d filler word(s): %s",
                        title, n_fillers,
                        [w.text for w in all_clip_words
                         if w.text.lower().strip(".,!?…") in _TRIM_FILLER_WORDS])

        if not clip_words:
            logger.warning("  Clip '%s' has no words after filler removal — dropping", title)
            continue

        new_start = start
        new_end = end

        # Trim leading silence
        if clip_words[0].start - start > _SILENCE_TRIM_THRESHOLD:
            new_start = clip_words[0].start

        # Trim trailing silence
        if end - clip_words[-1].end > _SILENCE_TRIM_THRESHOLD:
            new_end = clip_words[-1].end + 0.3   # small breath after last word

        # --- Build keep-segments by scanning internal gaps ---
        keep_segments: list[tuple[float, float]] = []
        seg_start = new_start
        for i in range(len(clip_words) - 1):
            gap = clip_words[i + 1].start - clip_words[i].end
            if gap > _SILENCE_TRIM_THRESHOLD:
                seg_end = clip_words[i].end + 0.15  # tiny breath after word
                keep_segments.append((seg_start, seg_end))
                seg_start = clip_words[i + 1].start
        keep_segments.append((seg_start, new_end))

        internal_removed = sum(
            keep_segments[i + 1][0] - keep_segments[i][1]
            for i in range(len(keep_segments) - 1)
        )
        after_duration = sum(e - s for s, e in keep_segments)
        total_removed = (end - start) - after_duration
        edge_removed = total_removed - internal_removed

        logger.info(
            "  Trimming silences for clip '%s' — before: %.1fs, after: %.1fs, removed: %.2fs"
            " (edge: %.2fs, internal gaps: %d × %.2fs total)",
            title, end - start, after_duration, total_removed,
            edge_removed, len(keep_segments) - 1, internal_removed,
        )

        if after_duration < CLIP_MIN_DURATION:
            logger.warning(
                "  Clip '%s' dropped after silence trim — duration %.1fs < %ds",
                title, after_duration, CLIP_MIN_DURATION,
            )
            continue

        trimmed.append({
            **c,
            "start": new_start,
            "end": new_end,
            "segments": keep_segments,
        })

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
                            "cy": bbox.origin_y + bbox.height // 2,
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
                    faces.append({"cx": int(x + w // 2), "cy": int(y + h // 2), "area": int(w * h)})
            frame_idx += 1
    finally:
        cap.release()

    return faces


def detect_speakers(video_path: str, sample_every_n_frames: int = 15) -> list[tuple[int, int]]:
    """
    Return (cx, cy) center positions of the up to two largest faces found in the video,
    sorted left to right. Returns [] (no faces), [(cx, cy)] (one speaker), or
    [(cx_left, cy_left), (cx_right, cy_right)] (two speakers).

    cy values are averaged across all detections for that speaker to give a stable
    vertical anchor for crop positioning.
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

    # Sort by area descending, pick up to 2 spatially distinct faces.
    # For each distinct speaker accumulate all cy samples and average them.
    MIN_X_SEP = 100  # px — faces closer than this are the same person
    speaker_cx: list[int] = []
    speaker_cy_sums: list[list[int]] = []

    for face in sorted(faces, key=lambda f: f["area"], reverse=True):
        matched = next(
            (i for i, sx in enumerate(speaker_cx) if abs(face["cx"] - sx) < MIN_X_SEP),
            None,
        )
        if matched is None:
            speaker_cx.append(face["cx"])
            speaker_cy_sums.append([face["cy"]])
        else:
            speaker_cy_sums[matched].append(face["cy"])
        if len(speaker_cx) == 2:
            break

    # Build final list sorted left-to-right with averaged cy
    raw = [
        (cx, int(sum(cy_list) / len(cy_list)))
        for cx, cy_list in zip(speaker_cx, speaker_cy_sums)
    ]
    result = sorted(raw, key=lambda p: p[0])
    logger.info(
        "detect_speakers: %d speaker(s) at (cx, cy)=%s in %s",
        len(result), result, video_path,
    )
    return result


# ---------------------------------------------------------------------------
# 4b. Per-clip face tracking (used in render step instead of full-video detection)
# ---------------------------------------------------------------------------

def _detect_face_track_on_clip(
    video_path: str,
    sample_every_n_frames: int = 30,
) -> list[tuple[float, int, int]]:
    """
    Run face detection on a trimmed clip every N frames.
    Returns [(timestamp_sec, cx, cy), ...] only for frames where a face was detected.
    Uses MediaPipe if available, falls back to OpenCV Haar cascade.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: list[tuple[float, int, int]] = []
    frame_idx = 0

    if _MEDIAPIPE_AVAILABLE:
        _ensure_face_model()
        detector = _mp_vision.FaceDetector.create_from_options(
            _mp_vision.FaceDetectorOptions(
                base_options=_mp_python.BaseOptions(model_asset_path=str(_FACE_MODEL_PATH)),
                min_detection_confidence=FACE_CONF_THRESHOLD,
            )
        )
    else:
        frontal_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n_frames == 0:
                t = frame_idx / fps
                if _MEDIAPIPE_AVAILABLE:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    det = detector.detect(mp_img)
                    if det.detections:
                        best = max(
                            det.detections,
                            key=lambda d: d.bounding_box.width * d.bounding_box.height,
                        )
                        bbox = best.bounding_box
                        results.append((
                            t,
                            bbox.origin_x + bbox.width // 2,
                            bbox.origin_y + bbox.height // 2,
                        ))
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    detected = frontal_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
                    )
                    if len(detected):
                        x, y, fw, fh = max(detected, key=lambda d: d[2] * d[3])
                        results.append((t, int(x + fw // 2), int(y + fh // 2)))
            frame_idx += 1
    finally:
        cap.release()
        if _MEDIAPIPE_AVAILABLE:
            detector.close()

    return results


def build_clip_crop_filter(
    video_path: str,
    source_w: int,
    source_h: int,
    focus_zone: str = "center",
) -> str:
    """
    Build a 9:16 crop+scale filter with smooth horizontal face tracking.

    - Samples every 30 frames of the already-trimmed clip
    - Smooths detected cx positions with a ±2 sample moving average
    - Builds a piecewise-linear x expression so the crop follows the face
    - Anchors the face at FACE_Y_TARGET_RATIO (25%) from the top vertically
    - Falls back to focus_zone-biased crop if no face is detected
    - Scales the source up before cropping when it is too narrow to fill 9:16
    """
    track = _detect_face_track_on_clip(video_path)
    logger.info("build_clip_crop_filter: %d face sample(s) in %s", len(track), video_path)

    # Crop window in exact 9:16; make both dimensions even (H.264 requirement).
    crop_h = int(source_h * FACE_CROP_H_RATIO) & ~1
    crop_w = (crop_h * 9 // 16) & ~1

    # If the source is too narrow to fill crop_w, scale it up first so the crop
    # always covers exactly crop_w × crop_h pixels with no black borders.
    if crop_w > source_w:
        scale_factor = crop_w / source_w
        scaled_h = int(source_h * scale_factor) & ~1
        pre_scale = f"scale={crop_w}:{scaled_h}:flags=lanczos,"
        eff_w, eff_h = crop_w, scaled_h
        track = [(t, int(cx * scale_factor), int(cy * scale_factor)) for t, cx, cy in track]
    else:
        pre_scale = ""
        eff_w, eff_h = source_w, source_h

    crop_h = min(crop_h, eff_h) & ~1

    def clamp_x(cx: int) -> int:
        return max(0, min(eff_w - crop_w, cx - crop_w // 2))

    def face_y(cy: int) -> int:
        y = cy - int(FACE_Y_TARGET_RATIO * crop_h)
        return max(0, min(eff_h - crop_h, y))

    if not track:
        if focus_zone == "left":
            x = 0
        elif focus_zone == "right":
            x = max(0, eff_w - crop_w)
        else:
            x = max(0, (eff_w - crop_w) // 2)
        logger.info("build_clip_crop_filter: no face → %s crop x=%d", focus_zone, x)
        return f"{pre_scale}crop={crop_w}:{crop_h}:{x}:0,scale={FRAME_W}:{FRAME_H}:flags=lanczos"

    # Stable vertical anchor: average cy across all samples
    avg_cy = int(sum(t[2] for t in track) / len(track))
    y = face_y(avg_cy)

    # Smooth horizontal positions with a ±2 sample moving average
    cx_raw = [t[1] for t in track]
    window = 2
    cx_smooth = []
    for i in range(len(cx_raw)):
        lo = max(0, i - window)
        hi = min(len(cx_raw), i + window + 1)
        cx_smooth.append(int(sum(cx_raw[lo:hi]) / (hi - lo)))

    xs = [clamp_x(cx) for cx in cx_smooth]
    times = [t[0] for t in track]

    if len(track) == 1:
        logger.info("build_clip_crop_filter: 1 sample → static crop x=%d y=%d", xs[0], y)
        return f"{pre_scale}crop={crop_w}:{crop_h}:{xs[0]}:{y},scale={FRAME_W}:{FRAME_H}:flags=lanczos"

    # Build piecewise-linear FFmpeg expression for x (uses \, to escape commas)
    # For each interval [t_i, t_{i+1}]: x_i + (x_{i+1}-x_i)*(t-t_i)/(t_{i+1}-t_i)
    expr = str(xs[-1])  # hold last value beyond final sample
    for i in range(len(times) - 2, -1, -1):
        t0, x0 = times[i], xs[i]
        t1, x1 = times[i + 1], xs[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        dx = x1 - x0
        if dx == 0:
            segment = str(x0)
        else:
            segment = f"({x0}+{dx}*(t-{t0:.3f})/{dt:.3f})"
        expr = f"if(lt(t\\,{t1:.3f})\\,{segment}\\,{expr})"

    logger.info("build_clip_crop_filter: tracking expr len=%d y=%d", len(expr), y)
    return f"{pre_scale}crop={crop_w}:{crop_h}:{expr}:{y},scale={FRAME_W}:{FRAME_H}:flags=lanczos"


# ---------------------------------------------------------------------------
# 5. Dynamic crop filter (speaker switching)
# ---------------------------------------------------------------------------

def _speaker_turn_times(
    words: list[Word],
    clip_start: float,
    clip_end: float,
) -> list[float]:
    """
    Return output-timeline timestamps where the speaker label changes,
    enforcing SPEAKER_SWITCH_MIN_INTERVAL between consecutive switches.
    Words must already be in output timeline (t=0 at clip start).
    Returns [] if no speaker info is present on the words.
    """
    clip_words = [w for w in words if clip_start <= w.start <= clip_end and w.speaker]
    if not clip_words:
        return []

    turn_times: list[float] = []
    last_switch = -SPEAKER_SWITCH_MIN_INTERVAL  # allow first switch from t=0
    current_speaker = clip_words[0].speaker

    for w in clip_words:
        if w.speaker != current_speaker:
            t = w.start - clip_start
            if t - last_switch >= SPEAKER_SWITCH_MIN_INTERVAL:
                turn_times.append(t)
                last_switch = t
            current_speaker = w.speaker  # always update, even if we didn't add the turn

    return turn_times


def build_dynamic_crop_filter(
    speaker_positions: list[tuple[int, int]],
    source_w: int,
    source_h: int,
    clip_start: float,
    clip_end: float,
    words: list[Word],
    focus_zone: str = "center",
) -> str:
    """
    Build an FFmpeg crop+scale filter string.

    Framing improvements vs. the old static crop:
    - Vertical: use FACE_CROP_H_RATIO of source_h as crop height, then shift crop_y
      so the detected face lands at FACE_Y_TARGET_RATIO from the top of the output
      frame (not dead-centre).  For two speakers the average cy is used so the frame
      doesn't jump vertically on speaker switches.
    - Horizontal: with two speakers, switch on every speaker-turn detected via
      WhisperX word-level diarization (word.speaker field), enforcing a minimum
      interval of SPEAKER_SWITCH_MIN_INTERVAL seconds.  Falls back to pause-based
      switching when no speaker labels are present.
    """
    # --- Crop dimensions: exact 9:16, even pixel count ---
    crop_h = int(source_h * FACE_CROP_H_RATIO) & ~1
    crop_w = (crop_h * 9 // 16) & ~1

    # Scale source up if it is too narrow to fill the 9:16 crop window.
    if crop_w > source_w:
        scale_factor = crop_w / source_w
        scaled_h = int(source_h * scale_factor) & ~1
        pre_scale = f"scale={crop_w}:{scaled_h}:flags=lanczos,"
        eff_w, eff_h = crop_w, scaled_h
        speaker_positions = [
            (int(cx * scale_factor), int(cy * scale_factor))
            for cx, cy in speaker_positions
        ]
    else:
        pre_scale = ""
        eff_w, eff_h = source_w, source_h

    crop_h = min(crop_h, eff_h) & ~1

    def clamp_x(cx: int) -> int:
        return max(0, min(eff_w - crop_w, cx - crop_w // 2))

    def face_crop_y(cy: int) -> int:
        y = cy - int(FACE_Y_TARGET_RATIO * crop_h)
        return max(0, min(eff_h - crop_h, y))

    # --- No faces: use focus_zone to position crop horizontally ---
    if not speaker_positions:
        if focus_zone == "left":
            nf_x = 0
        elif focus_zone == "right":
            nf_x = max(0, eff_w - crop_w)
        else:
            nf_x = max(0, (eff_w - crop_w) // 2)
        return f"{pre_scale}crop={crop_w}:{crop_h}:{nf_x}:0,scale={FRAME_W}:{FRAME_H}:flags=lanczos"

    # --- One speaker: static crop centred on face ---
    if len(speaker_positions) == 1:
        cx, cy = speaker_positions[0]
        x = clamp_x(cx)
        y = face_crop_y(cy)
        return f"{pre_scale}crop={crop_w}:{crop_h}:{x}:{y},scale={FRAME_W}:{FRAME_H}:flags=lanczos"

    # --- Two speakers ---
    (cx0, cy0), (cx1, cy1) = speaker_positions[0], speaker_positions[1]
    x0, x1 = clamp_x(cx0), clamp_x(cx1)
    avg_cy = (cy0 + cy1) // 2
    y = face_crop_y(avg_cy)

    switch_times = _speaker_turn_times(words, clip_start, clip_end)
    if switch_times:
        logger.info(
            "switch_crop: %d speaker turn(s) in clip [%.2f–%.2f] (diarization), "
            "x0=%d x1=%d y=%d",
            len(switch_times), clip_start, clip_end, x0, x1, y,
        )
    else:
        clip_words = [w for w in words if clip_start <= w.start <= clip_end]
        last_switch_t = -SPEAKER_SWITCH_MIN_INTERVAL
        for i in range(len(clip_words) - 1):
            gap = clip_words[i + 1].start - clip_words[i].end
            if gap >= SUB_SILENCE_GAP:
                t = clip_words[i].end - clip_start
                if t - last_switch_t >= SPEAKER_SWITCH_MIN_INTERVAL:
                    switch_times.append(t)
                    last_switch_t = t
        logger.info(
            "switch_crop: %d pause-based switch(es) in clip [%.2f–%.2f] (fallback), "
            "x0=%d x1=%d y=%d",
            len(switch_times), clip_start, clip_end, x0, x1, y,
        )

    if not switch_times:
        return f"{pre_scale}crop={crop_w}:{crop_h}:{x0}:{y},scale={FRAME_W}:{FRAME_H}:flags=lanczos"

    xs = [x0, x1]
    expr = str(xs[len(switch_times) % 2])
    for i, t in reversed(list(enumerate(switch_times))):
        expr = f"if(lt(t\\,{t:.3f})\\,{xs[i % 2]}\\,{expr})"

    return f"{pre_scale}crop={crop_w}:{crop_h}:{expr}:{y},scale={FRAME_W}:{FRAME_H}:flags=lanczos"


# ---------------------------------------------------------------------------
# 5b. Disk space guard
# ---------------------------------------------------------------------------

def _check_workspace_disk_space(min_gb: float = 5.0) -> None:
    """Raise RuntimeError if /workspace has less than min_gb free. Skip if path absent."""
    import shutil as _su
    workspace = "/workspace"
    try:
        usage = _su.disk_usage(workspace)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_gb:
            raise RuntimeError(
                f"Insufficient disk space on {workspace}: {free_gb:.1f} GB free, "
                f"need at least {min_gb:.0f} GB. Free up space and retry."
            )
        logger.info("Disk space check OK: %.1f GB free on %s", free_gb, workspace)
    except FileNotFoundError:
        logger.info("Disk space check skipped: %s not found (non-RunPod env)", workspace)


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


def build_title_card_filter_keo(title: str, duration: float) -> list[str]:
    """
    Keo style title card: clean white floating text, no box/pill, mixed case,
    Arial/Helvetica Neue, size 52, subtle drop shadow.
    """
    display_duration = min(3.0, duration)
    lines, fontsize = _wrap_title(title, MAX_TITLE_W, 52)
    line_height = int(fontsize * 1.4)
    card_top_y = SAFE_MARGIN + 60

    filters: list[str] = []
    for i, line in enumerate(lines):
        y = card_top_y + i * line_height
        filters.append(
            f"drawtext=fontfile={FONT_PATH_BOLD}"
            f":fontsize={fontsize}"
            f":fontcolor=white"
            f":shadowcolor=black@0.5"
            f":shadowx=2:shadowy=2"
            f":text='{_escape_drawtext(line)}'"
            f":x=(w-text_w)/2"
            f":y={y}"
            f":enable='between(t\\,0\\,{display_duration:.3f})'"
        )
    return filters


def build_title_card_filter_tovaritch(hook_text: str, duration: float) -> list[str]:
    """
    Tovaritch title card: hook text in Impact ALL CAPS, size 80, red (0xFF2200).
    Centered horizontally at 38% from top. No box, no pill, no outline.
    First line red, second line white if text wraps.
    """
    display_duration = min(3.0, duration)
    text = (hook_text or "").upper().strip()
    if not text:
        return []
    lines, fontsize = _wrap_title(text, MAX_TITLE_W, 80)
    line_height = int(fontsize * 1.35)
    card_top_y = int(FRAME_H * 0.38)

    colors = ["0xFF2200", "0xFFFFFF"]
    filters: list[str] = []
    for i, line in enumerate(lines):
        y = card_top_y + i * line_height
        color = colors[i % 2]
        filters.append(
            f"drawtext=fontfile={FONT_PATH_IMPACT}"
            f":fontsize={fontsize}"
            f":fontcolor={color}"
            f":text='{_escape_drawtext(line)}'"
            f":x=(w-text_w)/2"
            f":y={y}"
            f":box=0"
            f":borderw=0"
            f":enable='between(t\\,0\\,{display_duration:.3f})'"
        )
    return filters


# ---------------------------------------------------------------------------
# 7. Subtitle drawtext filters
# ---------------------------------------------------------------------------

def _escape_drawtext(text: str) -> str:
    """Sanitize text for FFmpeg drawtext text= option.

    Uses a replace/strip approach rather than backslash-escaping chains, which
    are fragile and produce hard-to-debug double-escaping bugs.

    Rules (applied in order):
      %   → %%      (FFmpeg variable expansion guard — must come first)
      '   → \u2019  (right single quote — preserves apostrophe appearance and
                     keeps the surrounding text='...' delimiters intact)
      ;   → stripped (filter-graph separator in filter_complex)
      :   → stripped (option separator; strip to avoid any parser ambiguity)
      \\  → stripped (no backslash escaping used in this approach)
      [   → stripped (stream-specifier syntax)
      ]   → stripped
      {   → stripped (FFmpeg template vars: %{pts})
      }   → stripped
    """
    text = text.replace("%", "%%")
    text = text.replace("'", "\u2019")
    for ch in ";:\\[]{}":
        text = text.replace(ch, "")
    return text


def build_subtitle_filters(cards: list[SubtitleCard]) -> list[str]:
    """
    Build FFmpeg drawtext filter fragments for all subtitle cards.
    Returns one drawtext filter per card (not joined), so callers can chunk
    them into batches before passing to FFmpeg.

    Style: white text, Montserrat Bold size 52, black outline 2px, mixed case.
    No per-word highlighting — single pass per card.
    """
    if not cards:
        return []

    sub_y = int(FRAME_H * SUB_Y_RATIO)
    filters: list[str] = []

    for card in cards:
        try:
            full_text = _escape_drawtext(card.text)
            t_start = f"{card.display_start:.3f}"
            t_end   = f"{card.display_end:.3f}"
            filters.append(
                f"drawtext=fontfile={FONT_PATH}"
                f":fontsize={SUB_FONTSIZE}"
                f":fontcolor=white"
                f":text='{full_text}'"
                f":x=(w-text_w)/2"
                f":y={sub_y}"
                f":bordercolor=black"
                f":borderw={SUB_BORDER_W}"
                f":enable='between(t\\,{t_start}\\,{t_end})'"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "build_subtitle_filters: skipping card %r (t=%.3f–%.3f) — %s",
                card.text, card.display_start, card.display_end, exc,
            )

    return filters


def _is_keo_filler(word: str) -> bool:
    return word.lower().strip(".,!?…\"'\u2019") in _KEO_FILLER_WORDS_FR


def build_subtitle_filters_keo(cards: list[SubtitleCard]) -> list[str]:
    """
    Keo style subtitles: per-word bold/regular, no box/outline, drop shadow,
    Arial, size 52, positioned at 80% from top.
    French function words → regular weight; nouns/verbs/numbers/adjectives → bold.
    """
    if not cards:
        return []

    sub_y = int(FRAME_H * SUB_Y_KEO)
    space_w = int(52 * 0.28)  # approximate space width at fontsize 52
    filters: list[str] = []

    for card in cards:
        t_start = f"{card.display_start:.3f}"
        t_end = f"{card.display_end:.3f}"
        words = card.words

        # Compute per-word font and width
        word_data: list[tuple[str, str, int]] = []
        for w in words:
            font = FONT_PATH_REGULAR if _is_keo_filler(w.text) else FONT_PATH_BOLD
            width = _measure_text_width(w.text, 52)
            word_data.append((w.text, font, width))

        total_w = sum(wd[2] for wd in word_data) + space_w * max(0, len(word_data) - 1)
        current_x = max(SAFE_MARGIN, (FRAME_W - total_w) // 2)

        for word_text, font_path, word_width in word_data:
            text_escaped = _escape_drawtext(word_text)
            filters.append(
                f"drawtext=fontfile={font_path}"
                f":fontsize=52"
                f":fontcolor=white"
                f":shadowcolor=black@0.5"
                f":shadowx=2:shadowy=2"
                f":text='{text_escaped}'"
                f":x={current_x}"
                f":y={sub_y}"
                f":enable='between(t\\,{t_start}\\,{t_end})'"
            )
            current_x += word_width + space_w

    return filters


def build_subtitle_filters_tovaritch(cards: list[SubtitleCard]) -> list[str]:
    """
    Tovaritch style subtitles: ALL CAPS Impact size 76.
    Alternating red (0xFF2200) / white per card; questions always red.
    Positioned at 42% from top. NO box, NO outline, NO border.
    """
    if not cards:
        return []

    sub_y = int(FRAME_H * 0.42)
    filters: list[str] = []

    for idx, card in enumerate(cards):
        t_start = f"{card.display_start:.3f}"
        t_end = f"{card.display_end:.3f}"
        is_question = card.text.strip().endswith("?")
        # Use hex literals so FFmpeg never misinterprets the colour
        color = "0xFF2200" if (idx % 2 == 0 or is_question) else "0xFFFFFF"
        text = _escape_drawtext(card.text.upper())
        filters.append(
            f"drawtext=fontfile={FONT_PATH_IMPACT}"
            f":fontsize=76"
            f":fontcolor={color}"
            f":text='{text}'"
            f":x=(w-text_w)/2"
            f":y={sub_y}"
            f":box=0"
            f":borderw=0"
            f":enable='between(t\\,{t_start}\\,{t_end})'"
        )

    return filters


# ---------------------------------------------------------------------------
# 8. Music selection
# ---------------------------------------------------------------------------

_HIGH_ENERGY_CONTENT = {
    "esclave", "libre", "problème", "choc", "shocking", "dette", "ruine",
    "trap", "piège", "urgent", "alerte", "vrai", "menteur", "arnaque",
}
_EMOTIONAL_CONTENT = {
    "dormais", "sacrifice", "famille", "survivre", "loyer", "vie", "rêve",
    "histoire", "témoignage", "vécu", "amour", "larmes", "difficile",
}
_HIGH_ENERGY_TRACK = {"trap", "bass", "dark", "hard"}
_EMOTIONAL_TRACK = {"slowed", "reverb", "chill", "ethereal"}


def _classify_clip_energy(title: str, hook: str) -> str:
    """Return 'high', 'emotional', or 'default' based on title and hook text."""
    combined = (title + " " + hook).lower()
    # Shocking stat signals: large numbers, percentages, currency amounts
    if any(tok in combined for tok in ("000", "€", "%", "100%", "million", "milliard")):
        return "high"
    words_in_text = set(combined.split())
    if words_in_text & _HIGH_ENERGY_CONTENT:
        return "high"
    if words_in_text & _EMOTIONAL_CONTENT:
        return "emotional"
    return "default"


def pick_music_track(title: str = "", hook: str = "") -> str | None:
    """Return a .mp3 matched to the clip's energy level, or None if no tracks."""
    tracks = list(MUSIC_DIR.glob("*.mp3"))
    if not tracks:
        logger.warning("No music tracks found in %s", MUSIC_DIR)
        return None

    energy = _classify_clip_energy(title, hook)

    def _keywords_in_name(path: Path, keywords: set[str]) -> bool:
        return any(k in path.stem.lower() for k in keywords)

    if energy == "high":
        preferred = [t for t in tracks if _keywords_in_name(t, _HIGH_ENERGY_TRACK)]
        reason = "high-energy clip → prefer trap/bass/dark/hard tracks"
    elif energy == "emotional":
        preferred = [t for t in tracks if _keywords_in_name(t, _EMOTIONAL_TRACK)]
        reason = "emotional/story clip → prefer slowed/reverb/chill/ethereal tracks"
    else:
        preferred = []
        reason = "default energy → random track"

    pool = preferred if preferred else tracks
    chosen = random.choice(pool)
    logger.info("Music selection — energy=%s, reason=%s, track=%s (pool size=%d)",
                energy, reason, chosen.name, len(pool))
    return str(chosen)


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


def probe_video_duration(video_path: str) -> float:
    """Return actual duration of a video file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, encoding='utf-8', errors='replace',
    )
    if result.returncode != 0 or not result.stdout.strip():
        # Fallback: read container duration
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, encoding='utf-8', errors='replace',
        )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


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
    detections: list[tuple[int, int]],
    output_dir: str,
    subtitle_style: str = "classic",
    focus_zone: str = "center",
) -> str:
    """
    Render one clip to disk.  Returns the path to the final .mp4 file.

    Steps:
      a) Trim source video to clip window
      b) Apply dynamic crop → 1080×1920
      c) Burn subtitle cards (style: classic / keo / tovaritch)
      d) Overlay title card
      e) Mix background music (ducked under speech; skipped for tovaritch)
      f) Append CTA outro
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _render_clip_sync, spec, source_video,
                                      source_w, source_h, detections, output_dir,
                                      subtitle_style, focus_zone)


def _render_clip_sync(
    spec: ClipSpec,
    source_video: str,
    source_w: int,
    source_h: int,
    detections: list[tuple[int, int]],
    output_dir: str,
    subtitle_style: str = "classic",
    focus_zone: str = "center",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"clip_{spec.clip_id}_")

    try:
        duration = spec.output_duration

        # --- Step A: trim (with optional internal silence concat) ---
        trimmed = os.path.join(tmp_dir, "trimmed.mp4")
        if len(spec.segments) > 1:
            # Build filter_complex that trims each keep-segment and concatenates them
            n = len(spec.segments)
            filter_parts: list[str] = []
            for i, (s, e) in enumerate(spec.segments):
                filter_parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]")
                filter_parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
            filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[v][a]")
            logger.info("  Clip %s: concat %d segments (internal silences removed)", spec.clip_id, n)
            _run_ffmpeg([
                "-i", source_video,
                "-filter_complex", ";".join(filter_parts),
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-y", trimmed,
            ], desc=f"trim+concat {spec.clip_id}")
        else:
            _run_ffmpeg([
                "-ss", str(spec.start),
                "-to", str(spec.end),
                "-i", source_video,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-y", trimmed,
            ], desc=f"trim {spec.clip_id}")

        # --- Step B: crop to 9:16 ---
        # Run face detection on the trimmed clip for accurate per-clip tracking.
        # This replaces the old full-video averaged detections with a smooth
        # horizontal follow that anchors the face at 25% from top.
        crop_filter = build_clip_crop_filter(trimmed, source_w, source_h, focus_zone=focus_zone)
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
            # Probe actual cropped duration — spec.output_duration is calculated
            # and can diverge from what FFmpeg actually rendered (rounding, segment
            # boundaries).  Using the wrong value makes between(t,0,X) fire for
            # fewer frames than expected or not at all on short clips.
            actual_duration = probe_video_duration(cropped)
            if abs(actual_duration - duration) > 0.1:
                logger.warning(
                    "Clip %s — cropped duration mismatch: spec=%.3fs, actual=%.3fs (using actual)",
                    spec.clip_id, duration, actual_duration,
                )
            else:
                logger.info(
                    "Clip %s — cropped duration: spec=%.3fs, actual=%.3fs",
                    spec.clip_id, duration, actual_duration,
                )
            title_display_duration = min(3.0, actual_duration)
            logger.info(
                "Clip %s — title card enable=between(t,0,%.3f) (actual_duration=%.3fs)",
                spec.clip_id, title_display_duration, actual_duration,
            )

            # Each drawtext filter contains commas inside between(t,X,Y), so we
            # keep them as a list and chunk the list — never split a joined string.
            # Title card filters go FIRST so they are always rendered in pass 1
            # against the clean cropped.mp4 with guaranteed t=0 PTS at frame 0.
            if subtitle_style == "keo":
                title_filters = build_title_card_filter_keo(spec.title, actual_duration)
                subtitle_filters = build_subtitle_filters_keo(spec.subtitle_cards)
            elif subtitle_style == "tovaritch":
                # For tovaritch the title card shows the hook (viral sentence),
                # falling back to spec.title if hook is empty.
                hook_text = spec.hook or spec.title
                title_filters = build_title_card_filter_tovaritch(hook_text, actual_duration)
                subtitle_filters = build_subtitle_filters_tovaritch(spec.subtitle_cards)
            else:  # "classic" and any unknown value
                title_filters = build_title_card_filter(spec.title, actual_duration)
                subtitle_filters = build_subtitle_filters(spec.subtitle_cards)
            logger.info(
                "Clip %s — title card: %d filter(s), subtitles: %d filter(s)",
                spec.clip_id, len(title_filters), len(subtitle_filters),
            )
            for i, f in enumerate(title_filters):
                logger.info("Clip %s — title filter[%d]: %s", spec.clip_id, i, f)

            all_drawtext: list[str] = title_filters + subtitle_filters

            # tovaritch/keo: single FFmpeg pass — avoids any multi-pass colour
            # drift and guarantees exactly one drawtext invocation per filter.
            if subtitle_style in ("tovaritch", "keo"):
                chunks = [all_drawtext] if all_drawtext else [["null"]]
            else:
                _CHUNK = 10
                chunks = [all_drawtext[i:i + _CHUNK] for i in range(0, len(all_drawtext), _CHUNK)]
                if not chunks:
                    chunks = [["null"]]
            logger.info(
                "Clip %s — subtitle render: %d drawtext filters → %d pass(es) "
                "(title card in pass 1 of %d)",
                spec.clip_id, len(all_drawtext), len(chunks), len(chunks),
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
        # Tovaritch style keeps original audio only — skip music entirely.
        music_track = None if subtitle_style == "tovaritch" else pick_music_track(title=spec.title, hook=spec.hook)
        if music_track:
            with_music = os.path.join(tmp_dir, "music.mp4")
            _run_ffmpeg([
                "-i", text_burned,
                "-stream_loop", "-1", "-i", music_track,
                "-filter_complex",
                (
                    "[0:a]volume=1.0[speech];"
                    "[1:a]volume=0.18,atrim=0:duration={dur}[music];"
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
        # Only "classic" gets the CTA outro. "tovaritch" and "keo" skip it entirely.
        final = os.path.join(output_dir, f"{spec.clip_id}.mp4")
        if subtitle_style not in ("tovaritch", "keo"):
            try:
                append_cta_fast(pre_cta, final)
            except Exception as exc:
                logger.warning("CTA append failed (%s), using clip without CTA: %s", spec.clip_id, exc)
                import shutil
                shutil.copy2(pre_cta, final)
        else:
            import shutil
            shutil.copy2(pre_cta, final)
            logger.info("Clip %s — CTA skipped (style=%s)", spec.clip_id, subtitle_style)

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

async def process_video_pipeline(
    project_id: str,
    db: AsyncIOMotorDatabase,
    subtitle_style: str = "classic",
    focus_zone: str = "center",
) -> None:
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
        # Pre-flight: abort early if /workspace is nearly full
        # ----------------------------------------------------------------
        _check_workspace_disk_space(min_gb=5.0)

        # ----------------------------------------------------------------
        # Startup disk cleanup — remove stale /tmp/clip_* dirs left by
        # any previously crashed or killed pipeline run.
        # ----------------------------------------------------------------
        import glob as _glob
        import shutil as _shutil
        _tmp_root = tempfile.gettempdir()
        for _stale in _glob.glob(os.path.join(_tmp_root, "clip_*")):
            if os.path.isdir(_stale):
                _shutil.rmtree(_stale, ignore_errors=True)
                logger.info("Cleaned up stale clip temp dir: %s", _stale)

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
                    speaker=w.get("speaker"),
                )
                for w in stored_words
            ]
            logger.info(
                "Retry mode: %d pending clips, %d words from cache",
                len(pending_clip_ids), len(words),
            )
        else:
            # ----------------------------------------------------------------
            # Step 1: Transcribe (skip if transcript already cached, e.g. after
            # manual-mode pre-transcription)
            # ----------------------------------------------------------------
            stored_transcript = doc.get("transcript_words", [])
            if stored_transcript and not is_retry:
                logger.info(
                    "Transcript already cached (%d words) — skipping WhisperX",
                    len(stored_transcript),
                )
                words = [
                    Word(
                        text=w["text"],
                        start=float(w["start"]),
                        end=float(w["end"]),
                        precise=w.get("precise", True),
                        speaker=w.get("speaker"),
                    )
                    for w in stored_transcript
                ]
            else:
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
                    {"text": w.text, "start": w.start, "end": w.end, "precise": w.precise, "speaker": w.speaker}
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
                segments: list[tuple[float, float]] = raw.get("segments", [])

                if segments:
                    # Words remapped to output timeline (gaps excised, t=0 at clip start)
                    all_clip_words = [w for w in words if start <= w.start <= end]
                    clip_words = _remap_words_to_output(all_clip_words, segments)
                    cards = build_subtitle_cards(clip_words, clip_start=0.0)
                else:
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
                    segments=segments,
                )
                clip_specs_map[cid] = spec
                clip_docs.append({
                    "id": cid,
                    "caption": raw.get("title", ""),
                    "hook": raw.get("hook", ""),
                    "start": start,
                    "end": end,
                    "segments": [[s, e] for s, e in segments],
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
                segments: list[tuple[float, float]] = [
                    (float(s), float(e)) for s, e in c.get("segments", [])
                ]
                if segments:
                    all_clip_words = [w for w in words if start <= w.start <= end]
                    clip_words = _remap_words_to_output(all_clip_words, segments)
                    cards = build_subtitle_cards(clip_words, clip_start=0.0)
                else:
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
                    segments=segments,
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
                        detections, output_dir, subtitle_style, focus_zone,
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


# ---------------------------------------------------------------------------
# Manual mode pipelines
# ---------------------------------------------------------------------------

async def transcribe_only_pipeline(project_id: str, db: AsyncIOMotorDatabase) -> None:
    """Run transcription only, then set status='transcribed' so the frontend
    can present the manual clip selection UI."""
    projects = db["projects"]

    async def set_progress(
        step: str, progress: float, details: str = "", status: str = "processing"
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

    try:
        doc = await projects.find_one({"id": project_id})
        if not doc:
            return

        source_video_path = doc.get("local_video_path", "")
        if not source_video_path or not os.path.isfile(source_video_path):
            await set_progress("error", 0, "Video file not found. Please re-upload.", status="error")
            return

        await set_progress("transcribing", 5, "Transcription en cours…")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
            audio_path = af.name
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, extract_audio, source_video_path, audio_path
            )
            words = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_audio, audio_path
            )
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        word_dicts = [
            {"text": w.text, "start": w.start, "end": w.end, "precise": w.precise}
            for w in words
        ]
        flat_transcript = " ".join(w.text for w in words)
        duration = await asyncio.get_event_loop().run_in_executor(
            None, probe_video_duration, source_video_path
        )
        await projects.update_one(
            {"id": project_id},
            {"$set": {
                "transcript_words": word_dicts,
                "transcript": flat_transcript,
                "duration": duration,
                "status": "transcribed",
                "processing_step": "transcribed",
                "processing_progress": 1.0,
                "processing_details": "Transcription terminée — sélectionnez vos clips.",
            }},
        )
        logger.info("transcribe_only_pipeline done for %s (%d words)", project_id, len(words))

    except Exception as exc:
        logger.exception("transcribe_only_pipeline failed for %s: %s", project_id, exc)
        await set_progress("error", 0, str(exc)[:300], status="error")


async def render_manual_pipeline(
    project_id: str,
    manual_clips: list[dict],
    db: AsyncIOMotorDatabase,
    subtitle_style: str = "classic",
    focus_zone: str = "center",
) -> None:
    """Render clips with manually specified boundaries.

    Each item in *manual_clips* must have:
        start_seconds: float
        end_seconds:   float
        title:         str
        hook_note:     str  (optional)
    """
    from storage import put_file as _put_file

    projects = db["projects"]

    async def set_progress(
        step: str, progress: float, details: str = "", status: str = "processing"
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
        doc = await projects.find_one({"id": project_id})
        if not doc:
            return

        source_video_path = doc.get("local_video_path", "")
        if not source_video_path or not os.path.isfile(source_video_path):
            await set_progress("error", 0, "Video file not found. Please re-upload.", status="error")
            return

        # Restore transcript words
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

        # Build raw clip list from manual input, then apply silence trimming
        raw_clips = [
            {
                "start": float(c["start_seconds"]),
                "end": float(c["end_seconds"]),
                "title": c.get("title", ""),
                "hook": c.get("hook_note", ""),
            }
            for c in manual_clips
        ]
        raw_clips = trim_clip_silences(raw_clips, words)
        if not raw_clips:
            await set_progress("error", 0, "All clips were too short after silence trimming.", status="error")
            return

        # Build ClipSpec objects and initial DB docs
        clip_specs_map: dict[str, ClipSpec] = {}
        clip_docs: list[dict] = []

        for raw in raw_clips:
            cid = str(uuid.uuid4())[:8]
            start = float(raw["start"])
            end = float(raw["end"])
            segments: list[tuple[float, float]] = raw.get("segments", [])

            if segments:
                all_clip_words = [w for w in words if start <= w.start <= end]
                clip_words = _remap_words_to_output(all_clip_words, segments)
                cards = build_subtitle_cards(clip_words, clip_start=0.0)
            else:
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
                segments=segments,
            )
            clip_specs_map[cid] = spec
            clip_docs.append({
                "id": cid,
                "caption": raw.get("title", ""),
                "hook": raw.get("hook", ""),
                "start": start,
                "end": end,
                "segments": [[s, e] for s, e in segments],
                "status": "pending",
                "storage_path": "",
                "error": "",
            })

        await projects.update_one({"id": project_id}, {"$set": {"short_clips": clip_docs}})

        # Face detection (shared across all clips)
        await set_progress("detecting_speakers", 10, "Détection des visages…")
        source_w, source_h = await asyncio.get_event_loop().run_in_executor(
            None, get_video_dimensions, source_video_path
        )
        detections = await asyncio.get_event_loop().run_in_executor(
            None, detect_speakers, source_video_path
        )

        # Render all clips in parallel
        await set_progress("rendering", 20, f"Rendu de {len(clip_specs_map)} clips…")
        output_dir = tempfile.mkdtemp(prefix=f"project_{project_id}_manual_")
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
                        detections, output_dir, subtitle_style, focus_zone,
                    )
                    if not os.path.exists(local_path):
                        raise FileNotFoundError(f"Rendered file missing: {local_path}")
                    if os.path.getsize(local_path) < 100_000:
                        raise ValueError(
                            f"Rendered file too small: {os.path.getsize(local_path)} bytes"
                        )
                    storage_path = f"{project_id}/{spec.clip_id}.mp4"
                    await asyncio.get_event_loop().run_in_executor(
                        None, _put_file, local_path, storage_path
                    )
                    done_count += 1
                    progress = 20 + int(done_count / total * 75)
                    await update_clip(spec.clip_id, status="done", storage_path=storage_path)
                    await set_progress("rendering", progress, f"{done_count}/{total} clips prêts")
                    logger.info("Manual clip %d/%d done: %s", idx + 1, total, spec.clip_id)
                except Exception as exc:
                    logger.exception("Manual clip %s failed: %s", spec.clip_id, exc)
                    await update_clip(spec.clip_id, status="error", error=str(exc)[:500])
                finally:
                    local = os.path.join(output_dir, f"{spec.clip_id}.mp4")
                    if os.path.exists(local):
                        os.unlink(local)

        await asyncio.gather(*[
            render_and_upload(spec, i)
            for i, spec in enumerate(clip_specs_map.values())
        ])

        # Final status
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
        logger.info("render_manual_pipeline complete — %d done, %d errors", n_done, n_err)

    except Exception as exc:
        logger.exception("render_manual_pipeline failed for %s: %s", project_id, exc)
        await set_progress("error", 0, str(exc)[:300], status="error")

    finally:
        if output_dir and os.path.isdir(output_dir):
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
