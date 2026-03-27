#!/bin/bash
set -e

echo "=== Marvin Backend Startup ==="

# 1. Install/upgrade dependencies
echo "[1/4] Installing Python dependencies..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install -r /app/backend/requirements.txt --quiet

# 2. Pre-download Whisper large model if not already cached
echo "[2/4] Checking Whisper model cache..."
python -c "
import whisper, os
cache = os.path.expanduser('~/.cache/whisper')
model_file = os.path.join(cache, 'large.pt')
if os.path.exists(model_file):
    print('  Whisper large model already cached — skipping download.')
else:
    print('  Downloading Whisper large model (~3GB)...')
    whisper.load_model('large')
    print('  Done.')
"

# 3. Create assets directory if needed
echo "[3/4] Ensuring assets directory exists..."
mkdir -p /app/backend/assets

# 4. Start the server
echo "[4/4] Starting uvicorn server on port 8000..."
cd /app/backend
exec uvicorn server:app --host 0.0.0.0 --port 8000
