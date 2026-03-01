# ── Voice Pipeline — App Dockerfile ──────────────────────────────────────────
#
# Build:  docker compose build app
# Run:    docker compose up app
#
# Uses a two-stage build:
#   1. builder  — installs all dependencies (large, disposable)
#   2. runtime  — copies only what's needed (smaller final image)
#
# Python version pinned to match your dev environment. Change if needed.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile Python packages:
#   libsndfile1-dev   — soundfile (audio file I/O)
#   portaudio19-dev   — sounddevice (requires PortAudio headers to compile)
#   espeak-ng         — phonemizer, espeakng-loader, kokoro-onnx (text-to-phoneme engine)
#   ffmpeg            — opus_ffmpeg_io.py (USE_FFMPEG_IO=1 path)
#   libopus-dev       — FFmpeg Opus codec support
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1-dev \
    portaudio19-dev \
    espeak-ng \
    ffmpeg \
    libopus-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install --timeout 300 --retries 5 -r requirements.txt

# spaCy model in a separate layer so a requirements.txt change doesn't
# re-download the model, and vice versa.
RUN pip install --no-cache-dir --prefix=/install --timeout 300 --retries 5 \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libs (must mirror what was compiled against in builder):
#   libsndfile1     — soundfile
#   libportaudio2   — sounddevice
#   espeak-ng       — phonemizer, espeakng-loader, kokoro-onnx
#   ffmpeg          — opus_ffmpeg_io.py subprocess pool (USE_FFMPEG_IO=1)
#   libopus-dev     — FFmpeg Opus codec support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libportaudio2 \
    espeak-ng \
    ffmpeg \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Non-root user — don't run as root in production
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Port your FastAPI app listens on (matches PORT in .env).
# Prometheus /metrics is served by FastAPI at /metrics on this same port —
# start_prometheus=False in main.py means no separate metrics server is started.
EXPOSE 8000

# Gunicorn runs fine on Linux (inside the container) even though
# you're on Windows — that's the whole point of this Dockerfile.
# -c gunicorn_conf.py  — uses your tuned config (workers, timeouts, post_fork etc.)
# app.endpoint.main:app — adjust this path to match your actual module structure
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.endpoint.main:app"]