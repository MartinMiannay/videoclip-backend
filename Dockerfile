# CUDA 11.8 + cuDNN — matches the torch cu118 wheels
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: Python, ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install PyTorch with CUDA 11.8 first (heavy layer, cache-friendly)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install app dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY . .

# Pre-download the Whisper large model so cold starts are fast
RUN python -c "import whisper; whisper.load_model('large')"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
