#!/bin/bash
set -e

echo "=== Marvin Backend Startup ==="

# 0. Write .env from RunPod environment variables
echo "[0/4] Writing .env file..."
cat > /app/backend/.env <<EOF
MONGO_URL=${MONGO_URL}
DB_NAME=${DB_NAME}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
EMERGENT_LLM_KEY=${EMERGENT_LLM_KEY}
CORS_ORIGINS=${CORS_ORIGINS}
EOF
echo "  .env written."

# 1. Install system deps + Python dependencies
echo "[1/4] Installing dependencies..."
apt-get install -y --no-install-recommends libgles2-mesa libgl1-mesa-glx 2>/dev/null || true
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
