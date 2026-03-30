#!/bin/bash
set -e

echo "=== Marvin Backend Startup ==="

# 0. Write .env from RunPod environment variables
echo "[0/5] Writing .env file..."
cat > /app/backend/.env <<EOF
MONGO_URL=${MONGO_URL}
DB_NAME=${DB_NAME}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
EMERGENT_LLM_KEY=${EMERGENT_LLM_KEY}
CORS_ORIGINS=${CORS_ORIGINS}
EOF
echo "  .env written."

# 1. Pull latest code from GitHub
echo "[1/5] Pulling latest code..."
if [ -z "${GITHUB_TOKEN}" ]; then
    echo "  ERROR: GITHUB_TOKEN env var is not set — cannot clone/pull"
    exit 1
fi
REPO_URL="https://${GITHUB_TOKEN}@github.com/MartinMiannay/videoclip-backend.git"
cd /app
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "  Repo exists, pulling..."
    git remote set-url origin "$REPO_URL"
    git pull origin master
    echo "  git pull OK — now on commit: $(git rev-parse HEAD)"
    echo "  Commit message: $(git log -1 --pretty=%s)"
else
    echo "  /app is not a git repo — cloning fresh..."
    cd /
    rm -rf /app
    git clone "$REPO_URL" /app
    echo "  Clone OK — on commit: $(git -C /app rev-parse HEAD)"
    echo "  Commit message: $(git -C /app log -1 --pretty=%s)"
fi

# 2. Install system deps + Python dependencies
echo "[2/5] Installing dependencies..."
apt-get install -y --no-install-recommends libgles2 libgl1 2>/dev/null || true
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install -r /app/backend/requirements.txt --quiet

# 3. Pre-download Whisper large model if not already cached
echo "[3/5] Checking Whisper model cache..."
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

# 4. Create assets directory if needed
echo "[4/5] Ensuring assets directory exists..."
mkdir -p /app/backend/assets

# 5. Kill any existing process on port 8000, then start fresh
echo "[5/5] Starting uvicorn server on port 8000..."
echo "  Checking for existing process on port 8000..."
fuser -k 8000/tcp 2>/dev/null && echo "  Killed existing process on port 8000." || echo "  Port 8000 is free."
cd /app/backend
exec uvicorn server:app --host 0.0.0.0 --port 8000
