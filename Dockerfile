# =============================================================================
# Stage 1: Builder - Install dependencies and download models
# =============================================================================
ARG DOCKER_FROM=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${DOCKER_FROM} AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    build-essential \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.8 (required for RTX 5090/Blackwell sm_120)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install flash-attention (optional, may fail on some systems)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Pre-download models to cache them in the image
RUN python -c "from qwen_tts import Qwen3TTSModel; Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')" || true
RUN python -c "import whisper; whisper.load_model('base')" || true

# =============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# =============================================================================
FROM ${DOCKER_FROM} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Qwen3-TTS Server"
LABEL description="Drop-in replacement for F5-TTS server using Qwen3-TTS"

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    libmagic1 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && git lfs install

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy cached models from builder (HuggingFace cache)
# Models are stored in /root/.cache/huggingface and /root/.cache/whisper
COPY --from=builder /root/.cache /root/.cache

ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

# Create necessary directories in /app (not /workspace which RunPod uses for network volumes)
RUN mkdir -p /app/server/outputs /app/server/resources

# Copy server files to /app
COPY server.py /app/server/
COPY start.sh /app/server/
COPY demo_speaker0.mp3 /app/server/resources/

# Fix line endings and make executable
RUN sed -i 's/\r$//' /app/server/start.sh \
    && chmod +x /app/server/start.sh

WORKDIR /app/server

# Expose the server port
EXPOSE 7860

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "/app/server/start.sh"]
