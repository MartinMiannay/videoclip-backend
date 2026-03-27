"""
Quick diagnostic: runs each pipeline step in isolation to find where WinError 2 occurs.
Usage: python test_pipeline.py <path_to_any_video_file>
"""
import sys, os, subprocess, tempfile, shutil, traceback
from pathlib import Path

VIDEO = sys.argv[1] if len(sys.argv) > 1 else None

print("=" * 60)
print("STEP 0: ffmpeg / ffprobe in PATH?")
print("  ffmpeg :", shutil.which("ffmpeg"))
print("  ffprobe:", shutil.which("ffprobe"))

print("\nSTEP 0b: direct subprocess call to ffmpeg -version")
try:
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    print("  returncode:", r.returncode)
    print("  first line:", r.stdout.splitlines()[0] if r.stdout else r.stderr[:80])
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()

if not VIDEO:
    print("\nNo video file provided — skipping file-based tests.")
    print("Usage: python test_pipeline.py <path_to_video>")
    sys.exit(0)

print(f"\nUsing video: {VIDEO}")
print(f"  exists: {os.path.exists(VIDEO)}")

print("\nSTEP 1: ffprobe get_video_dimensions")
try:
    r = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", VIDEO,
    ], capture_output=True, text=True)
    print("  returncode:", r.returncode)
    print("  stdout:", r.stdout.strip())
    print("  stderr:", r.stderr[:200] if r.stderr else "")
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()

print("\nSTEP 2: ffmpeg extract_audio")
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
    audio_path = af.name
print(f"  temp wav: {audio_path}")
try:
    r = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", VIDEO, "-vn", "-ac", "1", "-ar", "16000", "-y", audio_path,
    ], capture_output=True, text=True)
    print("  returncode:", r.returncode)
    if r.returncode == 0:
        print(f"  wav size: {os.path.getsize(audio_path):,} bytes")
    else:
        print("  stderr:", r.stderr[-400:])
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()

print("\nSTEP 3: whisper import + load model")
try:
    import whisper
    print("  whisper imported OK")
    print("  loading 'large' model (may take a while)...")
    model = whisper.load_model("large")
    print("  model loaded OK")
    print("  transcribing wav...")
    result = model.transcribe(audio_path, word_timestamps=True, language="fr", verbose=False)
    n_words = sum(len(s.get("words", [])) for s in result["segments"])
    print(f"  transcription OK — {n_words} words")
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()
finally:
    if os.path.exists(audio_path):
        os.unlink(audio_path)

print("\nSTEP 4: FONT_PATH check")
import sys as _sys
if _sys.platform == "win32":
    font = "C:/Windows/Fonts/arialbd.ttf"
    print(f"  font path: {font}")
    print(f"  exists: {os.path.exists(font)}")
else:
    print("  not Windows")

print("\nSTEP 5: ffmpeg drawtext with font path (simulated filter)")
try:
    font_escaped = "C\\:/Windows/Fonts/arialbd.ttf"
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        out_path = tf.name
    r = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", VIDEO,
        "-vf", f"drawtext=fontfile={font_escaped}:text='TEST':fontsize=40:fontcolor=white",
        "-t", "2", "-y", out_path,
    ], capture_output=True, text=True)
    print("  returncode:", r.returncode)
    if r.returncode != 0:
        print("  stderr:", r.stderr[-400:])
    else:
        print(f"  output size: {os.path.getsize(out_path):,} bytes — OK")
    os.unlink(out_path)
except Exception as e:
    print("  FAILED:", e)
    traceback.print_exc()

print("\nSTEP 6: music dir check")
music_dir = Path(__file__).parent / "music"
tracks = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.m4a"))
print(f"  music dir: {music_dir}")
print(f"  tracks found: {len(tracks)}")
if tracks:
    print(f"  first track: {tracks[0].name}")

print("\nSTEP 7: CTA files")
assets_dir = Path(__file__).parent / "assets"
cta = assets_dir / "cta_outro.mov"
cta_pre = assets_dir / "cta_preencoded.mp4"
print(f"  cta_outro.mov exists: {cta.exists()}")
print(f"  cta_preencoded.mp4 exists: {cta_pre.exists()}")

print("\n" + "=" * 60)
print("Diagnostics complete.")
